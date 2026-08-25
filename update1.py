$seeds=@(); while($seeds.Count -lt 100){$seeds+=Get-Random -Minimum 100 -Maximum 50000;$seeds=@($seeds|Sort-Object -Unique)}; foreach($seed in $seeds){Write-Host "Testing seed: $seed"; python client.py --server http://localhost:8000 --seed $seed --text "[S1] Hello. Thank you for calling Inspira Financial. How can I help you today? [S2] Hello. I need assistance with my account. Could you please help me with that ?" --output "seed_$seed.wav"}

python client.py --server http://localhost:8000 --text-file full_call.txt --reference-audio reference.wav --reference-text reference.txt --max-new-tokens 4096 --output full_call.wav --play


================================================================================
REFERENCE-CONDITIONED REQUEST
================================================================================
Request ID        : 5ba0ea51-cd77-4f6c-925f-d4008963800f
Blocks            : 5
Reference duration: 6.82s
Max new tokens    : 4096
================================================================================

--------------------------------------------------------------------------------
Block 1/5
[S2] Hello [S1] Hello Thank you for calling Inspira Financial For this call, would you prefer to continue in English or Spanish language? [S2] English please. [S1] Thank you Your language preference for this call is set to English How can I help you today? [S2] I need to verify my identity.
Inference         : 24557.90 ms
Audio duration    : 11.98 sec

--------------------------------------------------------------------------------
Block 2/5
[S1] I can help you with that To get started, I'll need to verify your identity. Can you please provide your four digit member ID? [S2] My member ID is 2043. [S1] I have two four Could you please provide the remaining two digits of your four digit member ID? [S2] My member ID is 2043.
Inference         : 27213.80 ms
Audio duration    : 13.34 sec

--------------------------------------------------------------------------------
Block 3/5
[S1] I have two zero four three. Is that correct? [S2] Yes, that is correct. [S1] Thank you. Now please provide the last four digits of your Social Security number. [S2] My Social Security number ends in 1234. [S1] I have one two three four. Is that correct? [S2] Yes, that is correct.
Inference         : 25564.94 ms
Audio duration    : 12.40 sec

--------------------------------------------------------------------------------
Block 4/5
[S1] Okay. Let me verify. Thank you. Your identity has been verified. How can I help you with your login today? [S2] Thank you for confirming my identity. [S1] You're welcome. Now that your identity is confirmed, how can I assist you with your login today? [S2] No further queries, ending this call now.
Inference         : 30415.91 ms
Audio duration    : 14.84 sec

--------------------------------------------------------------------------------
Block 5/5
[S1] Glad to help. Thank you. Have a nice day. [S2] No further queries, ending this call now.
Inference         : 7702.08 ms
Audio duration    : 3.58 sec

================================================================================
SERVER TOTAL
================================================================================
Blocks            : 5
Preprocess        : 20864.50 ms
Inference         : 115454.63 ms
Decode            : 134512.23 ms
SERVER TOTAL      : 270839.55 ms
Audio duration    : 56.13 sec
Generation RTF    : 2.0568
================================================================================
