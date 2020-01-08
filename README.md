# Use this code to render nice reports from fantasy stats

## Installation and running
- Fork repo.
- Enable github pages in settings.
- Use `pip install -r requirements.txt` to install required packages.
- Download chromedriver and put it [properly](https://stackoverflow.com/questions/42478591/python-selenium-chrome-webdriver).
- Look through `config_example` and create your own `.config` file with basketball and hockey leagues.
- Run `main.sh` or `src/main.py` if you don't want to publish them.
- Add `main.sh` to crontab (run at 14:00 GMT+3 each day) and enjoy!
