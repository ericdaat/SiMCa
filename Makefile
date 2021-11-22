venv: venv/bin/activate

install: requirements.txt
	test -d venv || virtualenv venv
	. venv/bin/activate; pip install -Ur requirements.txt
	touch venv/bin/activate

pytest: venv
	. venv/bin/activate && \
	pytest tests

widget: venv
	. venv/bin/activate && \
	FLASK_APP="src.widget.app" FLASK_DEBUG="True" flask run
