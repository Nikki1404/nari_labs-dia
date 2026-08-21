(venv) PS C:\Users\re_nikitav\Documents\nari_labs-dia>  python client.py --server http://localhost:8000  --text "[S1]Hello. [S2]Hello. Thank you for calling Inspira Financial. For this call, would you prefer to continue in English or Spanish language? [S1] English. [S2] Thank you, Your language preference for this call is set to English. How can I help you today? [S1] I need to verify my identity [S2] I can help you with that" --output hello.wav --play


for this it is not generating transcript after How can I help you today
so why it is breaking in mid and in config I would like to set s1 as female contant voice shouldn't change alsways and for speaker s2 i should be able to change voice is male or female via client side how to do that 
