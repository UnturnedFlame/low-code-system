import os
from django.core.management import execute_from_command_line

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'Backend_v2/settings.py')
execute_from_command_line(['manage.py', 'runserver', 'localhost:8000'])
