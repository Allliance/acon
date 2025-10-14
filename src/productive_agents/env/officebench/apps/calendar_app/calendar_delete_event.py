import os
import fire
import sys
from icalendar import Calendar, Event
from datetime import datetime
from productive_agents.env.officebench.apps import APPS_ROOT
import sys
from productive_agents.env.officebench.apps.calendar_app.calendar_list_events import list_events

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

DEMO = (
    'delete an event from a user\'s calendar given the event summary:'
    '{"app": "calendar", "action": "delete_event", "user": [USER_NAME], "summary": [EVENT_SUMMARY]}'
)


def construct_action(work_dir, args: dict, py_file_path=f'{APPS_ROOT}/calendar_app/calendar_delete_event.py'):
    return f'python3 {py_file_path} --user {args["user"]} --summary "{args["summary"]}"'


def delete_event(user, summary, workdir=None):
    try:
        calendar_workdir = f"{workdir}/testbed/calendar" if workdir is not None else '/testbed/calendar'
        calendar_file = f'{calendar_workdir}/{user}.ics'
        calendar = Calendar.from_ical(open(calendar_file, 'rb').read())

        for component in calendar.walk():
            if component.name == "VEVENT":
                if component.get('summary') == summary:
                    calendar.subcomponents.remove(component)
                    break
        with open(calendar_file, 'wb') as f:
            f.write(calendar.to_ical())

        content = list_events(user, workdir=workdir)  # List events to ensure the event is added
        return True, content
    except Exception as e:
        print('!!!', e)
        return False, f"Error deleting event: {e}."


def main(user, summary, workdir=None):
    success, content = delete_event(user, summary, workdir=workdir)
    if success:
        observation = f"OBSERVATION: Successfully delete an event named {summary} from {user}'s calendar."
        observation += f'\nThe current content of the {user}\'s calendar after the delete action is:\n{content}'
    else:
        observation = f"OBSERVATION: Failed to delete an event named {summary} from {user}'s calendar."
    return observation


if __name__ == '__main__':
    fire.Fire(main)