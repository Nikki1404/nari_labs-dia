$seeds=@(); while($seeds.Count -lt 100){$seeds+=Get-Random -Minimum 100 -Maximum 50000;$seeds=@($seeds|Sort-Object -Unique)}; foreach($seed in $seeds){Write-Host "Testing seed: $seed"; python client.py --server http://localhost:8000 --seed $seed --text "[S1] Hello. Thank you for calling Inspira Financial. How can I help you today? [S2] Hello. I need assistance with my account. Could you please help me with that ?" --output "seed_$seed.wav"}

python client.py --server http://localhost:8000 --text-file full_call.txt --reference-audio reference.wav --reference-text reference.txt --max-new-tokens 4096 --output full_call.wav --play


why gettingbthis 

[PLAY] Starting block 3/9
[PLAY] Finished block 3/9
Traceback (most recent call last):
  File "C:\Program Files\Python313\Lib\http\client.py", line 579, in _get_chunk_left
    chunk_left = self._read_next_chunk_size()
  File "C:\Program Files\Python313\Lib\http\client.py", line 546, in _read_next_chunk_size
    return int(line, 16)
ValueError: invalid literal for int() with base 16: b''

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "C:\Program Files\Python313\Lib\http\client.py", line 597, in _read_chunked
    while (chunk_left := self._get_chunk_left()) is not None:
                         ~~~~~~~~~~~~~~~~~~~~^^
  File "C:\Program Files\Python313\Lib\http\client.py", line 581, in _get_chunk_left
    raise IncompleteRead(b'')
http.client.IncompleteRead: IncompleteRead(0 bytes read)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\re_nikitav\Documents\nari_labs-dia\venv\Lib\site-packages\urllib3\response.py", line 905, in _error_catcher
    yield
  File "C:\Users\re_nikitav\Documents\nari_labs-dia\venv\Lib\site-packages\urllib3\response.py", line 1022, in _raw_read
    data = self._fp_read(amt, read1=read1) if not fp_closed else b""
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
  File "C:\Users\re_nikitav\Documents\nari_labs-dia\venv\Lib\site-packages\urllib3\response.py", line 1005, in _fp_read
    return self._fp.read(amt) if amt is not None else self._fp.read()
           ~~~~~~~~~~~~~^^^^^
  File "C:\Program Files\Python313\Lib\http\client.py", line 473, in read
    return self._read_chunked(amt)
           ~~~~~~~~~~~~~~~~~~^^^^^
  File "C:\Program Files\Python313\Lib\http\client.py", line 609, in _read_chunked
    raise IncompleteRead(b''.join(value)) from exc
http.client.IncompleteRead: IncompleteRead(0 bytes read)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\re_nikitav\Documents\nari_labs-dia\client_updated.py", line 1218, in <module>
    main()
    ~~~~^^
  File "C:\Users\re_nikitav\Documents\nari_labs-dia\client_updated.py", line 1167, in main
    run_reference_mode(
    ~~~~~~~~~~~~~~~~~~^
        args
        ^^^^
    )
    ^
  File "C:\Users\re_nikitav\Documents\nari_labs-dia\client_updated.py", line 773, in run_reference_mode
    read_frame(
    ~~~~~~~~~~^
        response.raw
        ^^^^^^^^^^^^
    )
    ^
  File "C:\Users\re_nikitav\Documents\nari_labs-dia\client_updated.py", line 87, in read_frame
    receive_exact(
    ~~~~~~~~~~~~~^
        raw,
        ^^^^
        4,
        ^^
    ),
    ^
  File "C:\Users\re_nikitav\Documents\nari_labs-dia\client_updated.py", line 68, in receive_exact
    part = raw.read(
        size - len(data)
    )
  File "C:\Users\re_nikitav\Documents\nari_labs-dia\venv\Lib\site-packages\urllib3\response.py", line 1110, in read
    data = self._raw_read(amt)
  File "C:\Users\re_nikitav\Documents\nari_labs-dia\venv\Lib\site-packages\urllib3\response.py", line 1021, in _raw_read
    with self._error_catcher():
         ~~~~~~~~~~~~~~~~~~~^^
  File "C:\Program Files\Python313\Lib\contextlib.py", line 162, in __exit__
    self.gen.throw(value)
    ~~~~~~~~~~~~~~^^^^^^^
  File "C:\Users\re_nikitav\Documents\nari_labs-dia\venv\Lib\site-packages\urllib3\response.py", line 928, in _error_catcher
    raise ProtocolError(f"Connection broken: {e!r}", e) from e
urllib3.exceptions.ProtocolError: ('Connection broken: IncompleteRead(0 bytes read)', IncompleteRead(0 bytes read))
