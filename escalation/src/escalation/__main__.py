"""Package entry point: python -m escalation submit ...

Delegates to escalation.submit.main() which uses an argparse 'submit'
subparser so that the leading 'submit' token (from sys.argv[1:]) is
consumed uniformly regardless of whether the package form or the
console-script form is used.
"""

from escalation.submit import main

raise SystemExit(main())
