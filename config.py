import configparser
import re


class Config:
    def __init__(self, filename, section="DEFAULT"):
        self.parser = configparser.ConfigParser()
        self.parser.read(filename)
        if section not in self.parser:
            raise ValueError(
                f"Section \"{section}\" not found in the configuration file."
            )
        self.load_config(section)

    def load_config(self, section):
        for key in self.parser[section]:
            value = self.parser.get(section, key)
            setattr(self, key, self.auto_type(value))

    def auto_type(self, value):

        if re.match(r"^[\s\d.,-]+$", value) and "," in value:
            parts = value.split(",")
            try:
                return tuple(
                    float(part.strip()) if "." in part else int(part.strip())
                    for part in parts
                )
            except ValueError:
                pass

        if "," in value and not re.match(r"^-?\d+(\.\d+)?$", value):
            return tuple(part.strip() for part in value.split(","))

        try:
            return int(value)
        except ValueError:
            pass

        try:
            return float(value)
        except ValueError:
            pass

        if value.lower() in ["true", "yes", "on"]:
            return True
        if value.lower() in ["false", "no", "off"]:
            return False

        return value
