# Everyday commands. `make` alone shows this list.
.PHONY: help install update uninstall app dmg backend dev test run logs

help:
	@echo "make install    one-command install: model server + app in /Applications (re-run to update)"
	@echo "make uninstall  remove app, settings, permissions (asks before deleting models/recordings)"
	@echo "make run        open the installed app"
	@echo "make logs       follow the model server log while dictating"
	@echo "make app        build build/Hushkey.app only"
	@echo "make dmg        build build/Hushkey.dmg for a GitHub release"
	@echo "make backend    (re)install only the model server"
	@echo "make dev        developer setup: editable backend in backend/.venv"
	@echo "make test       backend test suite (no weights needed)"

install update:
	./install.sh

uninstall:
	./uninstall.sh

run:
	open -b dev.hushkey.app

logs:
	@echo "Server output is shown in Hushkey ▸ Settings ▸ Backend. To run it in this terminal instead:"
	@echo '  quit Hushkey, then:  "$$HOME/Library/Application Support/Hushkey/venv/bin/python" -m hushkey.cli serve'

app:
	scripts/build_app.sh release

dmg:
	scripts/package_dmg.sh

backend:
	scripts/install_backend.sh

dev:
	scripts/setup_backend.sh

test:
	cd backend && .venv/bin/python -m pytest -q
