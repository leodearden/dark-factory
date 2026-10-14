// The jsdom render harness's own contract (dom_harness.mjs), starting with the
// toolchain identity guard: the harness must run the exact React bytes the
// browser runs, or it lies about what the browser renders.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { REPO_ROOT, cdnScriptTag, sha384Digest } from './_served_bundle.mjs';

const ROOT_PACKAGE_JSON = path.join(REPO_ROOT, 'package.json');
const REACT_PACKAGES = ['react', 'react-dom'];
const EXACT_VERSION_RE = /^\d+\.\d+\.\d+$/;

function devDependencies() {
  return JSON.parse(fs.readFileSync(ROOT_PACKAGE_JSON, 'utf8')).devDependencies ?? {};
}

function reactInstallCommand() {
  const specs = REACT_PACKAGES.map(pkg => `${pkg}@${cdnScriptTag(pkg).version}`);
  return `npm install --save-dev --save-exact ${specs.join(' ')}`;
}

function installedCopyOfCdnFile(pkg) {
  const requireFromRoot = createRequire(ROOT_PACKAGE_JSON);
  let packageJson;
  try {
    packageJson = requireFromRoot.resolve(`${pkg}/package.json`);
  } catch {
    assert.fail(`${pkg} is not installed under ${REPO_ROOT}/node_modules — run \`npm ci --no-audit --no-fund\` at the repo root`);
  }
  return path.join(path.dirname(packageJson), cdnScriptTag(pkg).pathInPackage);
}

for (const pkg of REACT_PACKAGES) {
  test(`package.json pins ${pkg} to exactly the version index.html loads`, () => {
    assert.equal(
      devDependencies()[pkg],
      cdnScriptTag(pkg).version,
      `${ROOT_PACKAGE_JSON} must pin devDependencies.${pkg} to the bare version index.html loads from the CDN. ` +
        `Remedy: ${reactInstallCommand()}`,
    );
  });

  test(`the installed ${pkg} build is byte-identical to the one index.html loads`, () => {
    const { url, integrity } = cdnScriptTag(pkg);
    const installed = installedCopyOfCdnFile(pkg);
    assert.equal(
      sha384Digest(installed),
      integrity,
      `${installed} is not the build index.html loads (${url}), so the harness would render with a ` +
        `different React than the browser. Remedy: ${reactInstallCommand()}, then \`npm ci\``,
    );
  });
}

test('package.json pins jsdom to a bare exact version', () => {
  assert.match(
    devDependencies().jsdom ?? 'absent',
    EXACT_VERSION_RE,
    `${ROOT_PACKAGE_JSON} must pin devDependencies.jsdom to a bare X.Y.Z. ` +
      'Remedy: npm install --save-dev --save-exact jsdom@<version>',
  );
});
