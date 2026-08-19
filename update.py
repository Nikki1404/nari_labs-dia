/usr/local/lib/python3.10/dist-packages/transformers/tokenization_utils_base.py:2355: UserWarning: `max_length` is ignored when `padding`=`True` and there is no truncation strategy. To pad to max length, use `padding='max_length'`.
  warnings.warn(
INFO:     172.17.0.1:46786 - "POST /tts HTTP/1.1" 500 Internal Server Error

(venv) PS C:\Users\re_nikitav\Documents\nari_labs-dia>  python client.py --server http://localhost:8000  --text "[S1] Hello. This is Dia running from my standalone API." --output hello.wav
Request failed (500): {"detail":"TTS generation failed: Failed to find C compiler. Please specify via CC environment variable or set triton.knobs.build.impl."}
