import logging


class ParamLogger(logging.Logger):
    def _log(self, level, msg, args, **kwargs):
        extra = kwargs.pop("extra", {})
        extra.update(kwargs)
        super()._log(
            level,
            msg,
            args,
            exc_info=kwargs.get("exc_info"),
            extra=extra,
            stack_info=kwargs.get("stack_info", False),
        )


logging.setLoggerClass(ParamLogger)

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s [%(levelname)s] "
        "[User=%(user)s | Realm=%(realm)s] "
        "[System=%(system)s | Component=%(component)s] "
        "[Status=%(status)s] "
        "[Context=%(context)s] "
        "[From=%(source)s] "
        "%(message)s"
    ),
    datefmt="%Y-%m-%d %H:%M:%S %Z",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

# EXAMPLE USAGE:
# logger.info(
#     "Loading dataset from data/train.csv",
#     user="dolly",
#     realm="AuburnSSO",
#     system="MLPipeline",
#     component="DataLoader",
#     status="IN_PROGRESS",
#     context="DatasetRead",
#     source="filesystem"
# )
