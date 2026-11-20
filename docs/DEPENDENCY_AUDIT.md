# NOORIX-AI Dependency Audit

Run from the repository root:

~~~bash
python scripts/dependency_audit.py
~~~

The report is written to a timestamped folder under
`reports/dependencies/` in both JSON and Markdown formats. The command reads
`pyproject.toml` and local Python distribution metadata only; it does not
contact package indexes or install, update, or remove packages.

The report lists base dependencies and every optional extra, the declared
specifier, whether a matching distribution is installed in the current Python
environment, and whether the declared range has an upper bound. Optional
dependencies that are not selected for the current environment may correctly
be absent.

For a policy review that returns a non-zero exit code when any non-exact
requirement has no upper version bound:

~~~bash
python scripts/dependency_audit.py --strict-upper-bounds
~~~

This is a review gate, not an automatic mandate to pin versions. Upper bounds
can prevent compatible security and bug-fix updates; maintainers should select
a tested compatibility window and update it through a reviewed release process.

## Important limitations

- Installed-version presence does not prove that the installed version
  satisfies the declared specifier.
- This is not a vulnerability, malware, license, provenance, or transitive
  dependency scan.
- No lockfile or reproducible dependency resolution is generated.
- Use an independently maintained vulnerability scanner and review its
  findings before distribution. Do not treat this offline report as a
  security certification.
