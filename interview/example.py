def hello_world():
    print("Hello World!")


def test_hello_world(capsys):
    hello_world()
    captured = capsys.readouterr()
    assert captured.out == "Hello World!\n"
