import configparser

config = configparser.ConfigParser()
config.read("config.ini")

PDF_PATH = config["PATHS"]["PDF_PATH"]
CHROMA_PATH = config["PATHS"]["CHROMA_PATH"]