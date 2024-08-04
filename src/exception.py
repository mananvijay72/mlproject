import sys


def get_error_message(error, error_detail: sys):
    # Retrieve exception information
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_message = str(error)

    # Create a detailed error message
    return f"Error occurred in script: {file_name}, line: {line_number}, message: {error_message}"

class CustomException(Exception):

    def __init__(self, error, error_detail: sys):
        # Initialize the base class with the error message
        super().__init__(str(error))

        # Store a detailed error message using the helper function
        self.error_message = get_error_message(error, error_detail)

    def __str__(self):
        # Return the detailed error message when the exception is printed
        return self.error_message

