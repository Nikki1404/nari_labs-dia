(venv) PS C:\Users\re_nikitav\Documents\nari_labs-dia>  python client.py --server http://localhost:8000  --text "[S1]Hello. [S2]Hello. Thank you for calling Inspira Financial. For this call, would you prefer to continue in English or Spanish language? [S1] English. [S2] Thank you, Your language preference for this call is set to English. How can I help you today? [S1] I need to verify my identity [S2] I can help you with that" --output hello.wav --play


python client.py --server http://localhost:8000 --text-file full_call.txt --seed 8472 --max-new-tokens 2048 --output full_call_seed_8472.wav --play
