import os
import json
import requests

class telegram:
    def __init__ (self, botname, token, useragent):
        self.botname = botname
        self.token = token
        self.savefile = "telegram.json"
        self.useragent = useragent

    def send_to_bot(self, access_point, data=None):
        headers = {'user-agent': self.useragent}
        try:
            r = requests.get('https://api.telegram.org/bot{0}/{1}'.format(self.token, access_point), data=data, timeout=40, headers=headers)
        except requests.exceptions.ConnectionError:
            return None
        except requests.exceptions.Timeout:
            return None
        return r

    def get_update(self):
        if self.data["last_update"] != 0:
            self.data["last_update"] += 1
        r = self.send_to_bot('getUpdates?timeout=30&offset={0}'.format(self.data["last_update"]))
        if not r:
            return None
        r_json = r.json()
        if not r_json["ok"]:
            return None
        if not "result" in r_json:
            return None
        if len(r_json["result"]) > 0:
            self.data["last_update"] = r_json["result"][-1]["update_id"]
        return r

    def open_session(self):
        if not self.get_telegram_data():
            self.data = {
                "last_update": 0
            }

    def close_session(self):
        self.set_telegram_data()

    def get_telegram_data(self):
        if not os.path.exists(self.savefile):
            return False
        else:
            #with open(self.savefile, encoding='utf-8') as data_file:
            with open(self.savefile) as data_file:
                try:
                    self.data = json.load(data_file)
                except:
                    raise
                    return False
        return True

    def set_telegram_data(self):
        #with open(self.savefile, 'w', encoding='utf-8') as data_file:
        with open(self.savefile, 'w') as data_file:
            json.dump(self.data, data_file, sort_keys=True, indent=4, separators=(',', ': '))
