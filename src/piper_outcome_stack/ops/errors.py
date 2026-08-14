"""Typed failures mapped to stable command-line exit codes."""


class OpsError(Exception):
    exit_code = 1


class ValidationError(OpsError):
    exit_code = 2


class IntegrityError(OpsError):
    exit_code = 3


class StateConflict(OpsError):
    exit_code = 4
