// The served bundle as node sees it: index.html's script tags, the Babel
// build it pins, and the top-level bindings each served script declares.
//
// A helper module, not a test: the `*.test.mjs` glob skips it. Its importers
// are classic_script_scope.test.mjs and test_datum_access_paths.py's
// client-binding census.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const TESTS_JS_DIR = path.dirname(fileURLToPath(import.meta.url));
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

const BABEL_TAG_RE =
  /<script\s+src="(https:\/\/unpkg\.com\/@babel\/standalone@[^"]+\/babel\.min\.js)"\s+integrity="(sha384-[^"]+)"/;

export function pinnedBabelTag() {
  const match = readIndexHtml().match(BABEL_TAG_RE);
  assert.ok(match, `found no @babel/standalone <script> tag with an integrity attribute in ${INDEX_HTML}`);
  return { url: match[1], integrity: match[2] };
}

function vendoredBabelDigest() {
  if (!fs.existsSync(VENDORED_BABEL)) return 'missing';
  return `sha384-${crypto.createHash('sha384').update(fs.readFileSync(VENDORED_BABEL)).digest('base64')}`;
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
