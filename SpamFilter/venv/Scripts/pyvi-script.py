#!C:\Users\Admin\PycharmProjects\SpamFilter\venv\Scripts\python.exe
# EASY-INSTALL-ENTRY-SCRIPT: 'pyvi==0.0.9.3','console_scripts','pyvi'
__requires__ = 'pyvi==0.0.9.3'
import re
import sys
from pkg_resources import load_entry_point

if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw?|\.exe)?$', '', sys.argv[0])
    sys.exit(
        load_entry_point('pyvi==0.0.9.3', 'console_scripts', 'pyvi')()
    )
