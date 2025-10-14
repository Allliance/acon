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
    'create a new event to a user\'s calendar where the time format is \'%Y-%m-%d %H:%M:%S\':'
    '{"app": "calendar", "action": "create_event", "user": [USER_NAME], "summary": [EVENT_SUMMARY], "time_start": [EVENT_START_TIME], "time_end": [EVENT_END_TIME]}'
)


def construct_action(word_dir, args: dict, py_file_path=f'{APPS_ROOT}/calendar_app/calendar_create_event.py'):
    # return f'python3 {py_file_path} --user {args["user"]} --summary "{args["summary"]}" --time_start "{args["time_start"]}" --time_end "{args["time_end"]}"'
    if isinstance(args["user"], list):
        args["user"] = 'Multiple users'
    return "python3 {} --user '''{}''' --summary '{}' --time_start '{}' --time_end '{}'".format(
        py_file_path, args["user"], args["summary"], args["time_start"], args["time_end"]
    )


def create_event(user, summary, time_start, time_end, workdir=None):
    print(workdir)
    calendar_workdir = f"{workdir}/testbed/calendar" if workdir is not None else '/testbed/calendar'
    print(calendar_workdir)
    os.makedirs(calendar_workdir, exist_ok=True)
    try:
        calendar_file = f'{calendar_workdir}/{user}.ics'
        if not os.path.exists(calendar_file):
            calendar = Calendar()
            calendar.add('prodid', '-//My Calendar Product//mxm.dk//')
            calendar.add('version', '2.0')
        else:
            calendar = Calendar.from_ical(open(calendar_file, 'rb').read())

        event = Event()
        event.add('summary', summary)
        event.add('dtstart', datetime.strptime(time_start, '%Y-%m-%d %H:%M:%S'))
        event.add('dtend', datetime.strptime(time_end, '%Y-%m-%d %H:%M:%S'))
        event.add('dtstamp', datetime.now())
        event.add('description', 'This is a test event')
        # event.add('location', 'Online')

        calendar.add_component(event)

        with open(calendar_file, 'wb') as f:
            f.write(calendar.to_ical())

        content = list_events(user, workdir=workdir)  # List events to ensure the event is added
        return True, content
    except Exception as e:
        print('!!!', e)
        return False, None


def main(user, summary, time_start, time_end, workdir=None):
    if user == 'Multiple users':
        observation = f"OBSERVATION: Failed to create a new event to {user}. Only support one user."
        return observation
    success, content = create_event(user, summary, time_start, time_end, workdir)
    if success:
        observation = f"OBSERVATION: Successfully create a new event to {user}'s calendar."
        observation += f'\nThe current content of the {user}\'s calendar after the create action is:\n{content}'
    else:
        observation = f"OBSERVATION: Failed to create a new event to {user}'s calendar."
    return observation


if __name__ == '__main__':
    fire.Fire(main)