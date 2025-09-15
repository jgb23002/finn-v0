from app.safety import is_oos

def test_oos_detection():
    assert is_oos("Should I take 50mg of ibuprofen?")
    assert is_oos("I have chest pain and shortness of breath.")
    assert not is_oos("How can I build a consistent sleep schedule?")
