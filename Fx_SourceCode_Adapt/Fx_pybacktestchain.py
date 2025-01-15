# create a function that calls all tests 

import pytest

def test_all():
    # pytest, folder is tests 
    pytest.main(['tests'])