TODO: Add a Note regarding need for `xz` module through `brew install xz` for the python interpreter to function
Resolution steps may include if using pyenv is to:
1. `brew install xz`
2. `pyenv uninstall <version>; pyenv install <version>`
3. `cd <project_directory>`
4. `rm -rf .venv; python -m venv .venv`
5. `pip install -r requirements.txt`
6. `source .venv/bin/activate`

TODO: For full reproducability to migrate to development workflow into Docker containers and use bind mounts properly 