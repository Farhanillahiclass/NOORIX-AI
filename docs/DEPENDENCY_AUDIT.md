# NOORIX-AI Dependency Audit

Run from the repository root:

~~~bash
python scripts/dependency_audit.py
~~~

The report is written to a timestamped folder under
`reports/dependencies/` in both JSON and Markdown formats. Report generation rejects
output paths containing symlinks before creating files. This reduces accidental
writes through linked directories but cannot eliminate concurrent filesystem
mutation races. The command reads
`pyproject.toml` and local Python distribution metadata only; it does not
contact package indexes or install, update, or remove packages.

The report lists base dependencies and every optional extra, the declared
specifier, whether a matching distribution is installed in the current Python
environment, whether the installed version satisfies its declared specifier (when `packaging` is installed), and whether the declared range has an upper bound. It rejects malformed version ranges, environment markers and direct-URL requirements rather than silently misclassifying them. Markdown table cells are escaped so unusual dependency metadata cannot create extra report rows. Optional
dependencies that are not selected for the current environment may correctly
be absent.

For a policy review that returns a non-zero exit code when any non-exact
requirement has no upper version bound:

~~~bash
python scripts/dependency_audit.py --strict-upper-bounds
~~~

The full validation environment installs `packaging` through the `dev` extra so version compatibility can be evaluated. In a minimal environment without `packaging`, compatibility is reported as unknown rather than guessed.

This is a review gate, not an automatic mandate to pin versions. Upper bounds
can prevent compatible security and bug-fix updates; maintainers should select
a tested compatibility window and update it through a reviewed release process.

## Important limitations

- Version compatibility is evaluated locally when `packaging` is installed; an unknown result is not a pass.
- This is not a vulnerability, malware, license, provenance, or transitive
  dependency scan.
- No lockfile or reproducible dependency resolution is generated.
- Use an independently maintained vulnerability scanner and review its
  findings before distribution. Do not treat this offline report as a
  security certification.


Report output uses a collision-safe timestamp directory: if the selected timestamp already exists, a numbered suffix is tried instead of overwriting it. JSON and Markdown files are opened in exclusive-create mode. Symlink components are rejected and rechecked before each report directory is created; as with other filesystem checks, concurrent path mutation cannot be eliminated completely.
