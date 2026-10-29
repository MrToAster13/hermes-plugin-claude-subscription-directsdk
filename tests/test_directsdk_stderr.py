"""A native child that dies without stream-json output surfaces its stderr tail in the error (ELI-681)."""
import io
import os
import sys

import pytest

import directsdk

REQUEST = dict(model='sonnet', messages=[{'role': 'user', 'content': 'fixture'}])


def _client(tmp_path, source):
    script = tmp_path / 'native.py'
    script.write_text(source)
    return directsdk.Client(command=[sys.executable, str(script)], env={'PATH': os.defpath, 'HOME': str(tmp_path)})


def _error(client, streaming):
    try:
        with pytest.raises(RuntimeError) as raised:
            result = client.create(**REQUEST, stream=streaming)
            if streaming:
                list(result)
    finally:
        client.close()
    return str(raised.value)


@pytest.mark.parametrize('streaming', [False, True])
def test_child_stderr_reaches_the_error_when_it_exits_without_output(tmp_path, streaming):
    client = _client(tmp_path, (
        'import sys\n'
        'sys.stdin.read()\n'
        'sys.stderr.write("error: unknown option --bogus-flag\\n")\n'
        'sys.exit(1)\n'))
    message = _error(client, streaming)
    assert message.startswith('Incomplete native response')
    assert 'error: unknown option --bogus-flag' in message


def test_large_child_stderr_keeps_only_a_bounded_tail(tmp_path):
    client = _client(tmp_path, (
        'import sys\n'
        'sys.stdin.read()\n'
        'for i in range(20000):\n'
        '    sys.stderr.write("noise line %06d\\n" % i)\n'
        'sys.stderr.write("LAST-LINE-MARKER\\n")\n'
        'sys.exit(1)\n'))
    message = _error(client, False)
    assert 'LAST-LINE-MARKER' in message
    assert 'noise line 000000' not in message
    assert len(message) < directsdk.STDERR_TAIL_BYTES + 500


def test_stderr_tail_memory_stays_bounded():
    tail = directsdk.StderrTail(io.BytesIO(b'x' * (5 * 1024 * 1024) + b'END'))
    tail.start()
    tail.join(5)
    assert tail.retained <= directsdk.STDERR_TAIL_BYTES
    assert tail.text().endswith('END')
    assert len(tail.text()) <= directsdk.STDERR_TAIL_BYTES


def test_silent_child_error_is_unchanged(tmp_path):
    client = _client(tmp_path, 'import sys\nsys.stdin.read()\nsys.exit(1)\n')
    assert _error(client, False) == 'Incomplete native response: assistant, message_stop and one result required'
