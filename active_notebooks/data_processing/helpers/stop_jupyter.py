class StopExecution(Exception):
    def _render_traceback_(self):
        return []
    
    
def stop():
    """Stops Jupyter notebook execution

    :raises StopExecution: to stop notebook execution
    """
    raise StopExecution
