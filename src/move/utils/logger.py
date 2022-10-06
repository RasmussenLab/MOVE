import logging
import os
import sys 

def get_logger(logging_path, file_name, script_name):
    
    # Creating the folder if doesn't exist
    isExist = os.path.exists(logging_path)
    if not isExist:
        os.makedirs(logging_path)
    
    # Logging to the file and console
    file_handler = logging.FileHandler(os.path.join(logging_path, file_name))
    #stream_handler = logging.StreamHandler()
    
    formatter = '%(levelname)-7s %(name)-10s   %(message)s'
    
    logging.basicConfig(level=logging.INFO, 
                        format=formatter,
    #                    handlers=[file_handler, stream_handler])
                        handlers=[file_handler])
    
    logger = logging.getLogger(script_name)

    logging.info('\n\n---------------- Starting running the script ---------------')
    return logger

