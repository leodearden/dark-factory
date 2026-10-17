// The served bundle as node sees it: index.html's script tags, the CDN builds
// it pins, how the browser compiles a text/babel tag, and the top-level
// bindings each served script declares.
//
// A helper module, not a test: the `*.test.mjs` glob skips it. It imports node
// built-ins only and does no work at import time, because
// test_datum_access_paths.py imports it from a bare `node -e` and
// classic_script_scope.test.mjs must run with no npm install.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const TESTS_JS_DIR = path.dirname(fileURLToPath(import.meta.url));
export const REPO_ROOT = path.resolve(TESTS_JS_DIR, '../../..');
export const REDUX_DIR = path.resolve(TESTS_JS_DIR, '../../src/dashboard/static/redux');
export const INDEX_HTML = path.join(REDUX_DIR, 'index.html');

export function readIndexHtml() {
  return fs.readFileSync(INDEX_HTML, 'utf8');
}

// Matches ONLY the local classic tags — `<script src="/static/redux/<name>.js?v=NN"></script>`.
// Excluded by construction:
//   - the `https://unpkg.com/...` CDN tags (path does not start with /static/redux/);
//   - the `type="text/babel"` .jsx tags (an attribute sits between `<script`
//     and `src`, and the path ends `.jsx`, not `.js`).
const CLASSIC_SCRIPT_RE = /<script\s+src="\/static\/redux\/([A-Za-z0-9_.-]+\.js)(?:\?[^"]*)?"\s*><\/script>/g;

export function classicScriptSrcs(html) {
  return [...html.matchAll(CLASSIC_SCRIPT_RE)].map(m => m[1]);
}

// Matches `<script type="text/babel" src="/static/redux/<name>.jsx?v=NN"></script>`,
// and the `text/jsx` spelling, which Babel-standalone executes too.
const BABEL_SCRIPT_RE =
  /<script\s+type="text\/(?:babel|jsx)"\s+src="\/static\/redux\/([A-Za-z0-9_.-]+\.jsx)(?:\?[^"]*)?"\s*><\/script>/g;

export function babelScriptSrcs(html) {
  return [...html.matchAll(BABEL_SCRIPT_RE)].map(m => m[1]);
}

// A vendored copy of the exact Babel build index.html loads, so no suite
// needs an npm install. Versionless on purpose: index.html's tag is the one
// pin, and loadVendoredBabel holds this copy to it.
export const VENDORED_BABEL = path.join(TESTS_JS_DIR, 'vendor', 'babel.min.js');

// Matches `<script src="https://unpkg.com/<pkg>@<version>/<path>" integrity="sha384-...">`,
// for a scoped (`@babel/standalone`) or unscoped (`react`) package name.
const CDN_SCRIPT_TAG_RE =
  /<script\s+src="(https:\/\/unpkg\.com\/((?:@[^/"@]+\/)?[^/"@]+)@([^/"]+)\/([^"]+))"\s+integrity="(sha384-[^"]+)"/g;

// The build index.html loads for `pkg`: its URL, the exact version, the file's
// path inside the npm package (so an installed copy can be found), and the SRI
// integrity the browser holds those bytes to.
export function cdnScriptTag(pkg) {
  const tag = [...readIndexHtml().matchAll(CDN_SCRIPT_TAG_RE)]
    .map(([, url, name, version, pathInPackage, integrity]) => ({ name, url, version, pathInPackage, integrity }))
    .find(({ name }) => name === pkg);
  assert.ok(tag, `found no ${pkg} <script> tag with an integrity attribute in ${INDEX_HTML}`);
  const { url, version, pathInPackage, integrity } = tag;
  return { url, version, pathInPackage, integrity };
}

export function pinnedBabelTag() {
  const { url, integrity } = cdnScriptTag('@babel/standalone');
  return { url, integrity };
}

// The SRI-format digest of `file` (`sha384-<base64>`), or 'missing' when there
// is no such file, so an identity check against an integrity attribute fails
// with its own remedy rather than an ENOENT.
export function sha384Digest(file) {
  if (!fs.existsSync(file)) return 'missing';
  return `sha384-${crypto.createHash('sha384').update(fs.readFileSync(file)).digest('base64')}`;
}

function vendoredBabelDigest() {
  return sha384Digest(VENDORED_BABEL);
}

// Loads the vendored Babel only once its sha384 matches index.html's pin, so a
// stale, truncated or missing copy fails with the refresh command rather than
// a parse error. require's own cache makes repeat calls cheap.
export function loadVendoredBabel() {
  const { url, integrity } = pinnedBabelTag();
  assert.equal(
    vendoredBabelDigest(),
    integrity,
    `${VENDORED_BABEL} is not the build index.html loads (${url}), so the text/babel ` +
      'scope test would compile the .jsx files with a different Babel than the browser. ' +
      `Refresh it: curl -sSfL -o ${VENDORED_BABEL} ${url}`,
  );
  return createRequire(import.meta.url)(VENDORED_BABEL);
}

// The options Babel-standalone builds for a text/babel tag with no
// `type="module"`, no data-presets/data-plugins and no data-targets. The
// bundle's script-tag loader keeps these literals through minification;
// classic_script_scope.test.mjs pins them to it.
export const SCRIPT_TAG_BABEL_OPTIONS = {
  presets: ['react', 'env'],
  plugins: ['transform-class-properties', 'transform-object-rest-spread', 'transform-flow-strip-types'],
  targets: { browsers: undefined },
};

// `source` compiled exactly as Babel-standalone compiles a text/babel tag for
// the browser: every top-level binding becomes a global `var`.
export function compileAsScriptTag(source, filename) {
  return loadVendoredBabel().transform(source, { ...SCRIPT_TAG_BABEL_OPTIONS, filename, sourceFileName: filename }).code;
}

function patternNames(pattern) {
  switch (pattern?.type) {
    case 'Identifier':
      return [pattern.name];
    case 'ObjectPattern':
      return pattern.properties.flatMap(prop => patternNames(prop.type === 'RestElement' ? prop.argument : prop.value));
    case 'ArrayPattern':
      return pattern.elements.flatMap(patternNames);
    case 'RestElement':
      return patternNames(pattern.argument);
    case 'AssignmentPattern':
      return patternNames(pattern.left);
    default:
      return [];
  }
}

function windowExportName(statement) {
  const expr = statement.expression;
  if (expr?.type !== 'AssignmentExpression') return null;
  const target = expr.left;
  const onWindow =
    target.type === 'MemberExpression' &&
    !target.computed &&
    target.object.type === 'Identifier' &&
    target.object.name === 'window';
  return onWindow ? target.property.name : null;
}

function statementBindings(statement) {
  switch (statement.type) {
    case 'FunctionDeclaration':
    case 'ClassDeclaration':
      return [statement.id.name];
    case 'VariableDeclaration':
      return statement.declarations.flatMap(decl => patternNames(decl.id));
    case 'ExpressionStatement': {
      const exported = windowExportName(statement);
      return exported === null ? [] : [exported];
    }
    default:
      return [];
  }
}

// Every name a served script binds at top level — a function, class, var,
// let or const, destructures included — plus every `window.<name> = ...` it
// assigns at top level, in source order. Only program.body is walked, so a
// name bound or assigned inside a function body is never reported.
export function topLevelBindings(source, filename) {
  const { ast } = loadVendoredBabel().transform(source, { presets: ['react'], ast: true, code: false, filename });
  return ast.program.body.flatMap(statementBindings);
}
