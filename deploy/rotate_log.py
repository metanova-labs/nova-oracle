"""Copy stdin to a log file, rotated at 50 MB with five old files kept."""
import logging
import sys
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(sys.argv[1], maxBytes=50 << 20, backupCount=5)
handler.setFormatter(logging.Formatter("%(message)s"))
for line in sys.stdin:
    handler.emit(logging.makeLogRecord({"msg": line.rstrip("\n")}))
