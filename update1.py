$seeds=@(); while($seeds.Count -lt 100){$seeds+=Get-Random -Minimum 100 -Maximum 50000;$seeds=@($seeds|Sort-Object -Unique)}; foreach($seed in $seeds){Write-Host "Testing seed: $seed"; python client.py --server http://localhost:8000 --seed $seed --text "[S1] Hello. Thank you for calling Inspira Financial. How can I help you today? [S2] Hello. I need assistance with my account. Could you please help me with that ?" --output "seed_$seed.wav"}

python client.py --server http://localhost:8000 --text-file full_call.txt --reference-audio reference.wav --reference-text reference.txt --max-new-tokens 4096 --output full_call.wav --play


(venv) PS C:\Users\re_nikitav\Documents\nari_labs-dia> python extract_voices.py --input-dir shortlisted\random_wavs --reference-text reference\reference.txt --agent-file shortlisted\random_wavs\seed_15423.wav --output-dir extracted_voices

================================================================================
REFERENCE
================================================================================
S1 / Agent text   : Hello. Thank you for calling Inspira Financial. How can I help you today?
S2 / Customer text: Hello. I need assistance with my account. Could you please help me with that?
================================================================================
[CUSTOMER] seed_121.wav -> customer_seed_121.wav | boundary=3.74s
[CUSTOMER] seed_1541.wav -> customer_seed_1541.wav | boundary=4.75s
[CUSTOMER] seed_15423.wav -> customer_seed_15423.wav | boundary=4.13s
[CUSTOMER] seed_15838.wav -> customer_seed_15838.wav | boundary=3.83s
[CUSTOMER] seed_19094.wav -> customer_seed_19094.wav | boundary=3.55s
[CUSTOMER] seed_21478.wav -> customer_seed_21478.wav | boundary=4.85s
[CUSTOMER] seed_29108.wav -> customer_seed_29108.wav | boundary=3.14s
[CUSTOMER] seed_33936.wav -> customer_seed_33936.wav | boundary=3.09s
[CUSTOMER] seed_45718.wav -> customer_seed_45718.wav | boundary=4.90s
[CUSTOMER] seed_687.wav -> customer_seed_687.wav | boundary=4.00s
Traceback (most recent call last):
  File "C:\Users\re_nikitav\Documents\nari_labs-dia\extract_voices.py", line 705, in <module>
    main()
    ~~~~^^
  File "C:\Users\re_nikitav\Documents\nari_labs-dia\extract_voices.py", line 646, in main
    raise FileNotFoundError(
    ...<2 lines>...
    )
FileNotFoundError: Agent WAV not found: shortlisted\random_wavs\shortlisted\random_wavs\seed_15423.wav

[S1] Hello. Thank you for calling Inspira Financial. How can I help you today? [S2] Hello. I need assistance with my account. Could you please help me with that?
