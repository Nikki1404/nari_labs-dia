(venv) PS C:\Users\re_nikitav\Documents\nari_labs-dia>  python client.py --server http://localhost:8000  --text "[S1]Hello. [S2]Hello. Thank you for calling Inspira Financial. For this call, would you prefer to continue in English or Spanish language? [S1] English. [S2] Thank you, Your language preference for this call is set to English. How can I help you today? [S1] I need to verify my identity [S2] I can help you with that" --output hello.wav --play


for this it is not generating transcript after How can I help you today
so why it is breaking in mid and in config I would like to set s1 as female contant voice shouldn't change alsways and for speaker s2 i should be able to change voice is male or female via client side how to do that 


i want to run this complete thing in single request 

 
[s1]Hello. Thank you for calling Inspira Financial. For this call, 
would you prefer to continue in English or Spanish language? 
[s2]English
[S1]Thank you.
Your language preference for this call is set to English. How can I help you today?
[S2]I need to verify my identity.
[S1]I can help you with that.
To get started, I'll need to verify your identity. Can you please provide your four digit member ID?
[S2]My member ID is 2043.
[s1]I have two four.Could you please provide the remaining two digits of your four digit member ID 
[S2]My member ID is 2043.
[S1]I have two zero four three. Is that correct?
[S2]Yes, that is correct. 
[S1]Thank you.
Now please provide the last four digits of your Social Security number.
[S2]My Social Security number ends in 1234. 
[S1]I have one two three four. Is that correct?
[S2]Yes, that is correct. 
[S1]Okay. Let me verify.
Thank you. Your identity has been verified. How can I help you with your login today?
[S2]Thank you for confirming my identity.
[S1]You're welcome. 
Now that your identity is confirmed, how can I assist you with your login today?
[S2]No further queries, ending this call now.
 [S1]Glad to help. Thank you. Have a nice day.

$seeds=@(); while($seeds.Count -lt 60){$seeds+=Get-Random -Minimum 100 -Maximum 25001;$seeds=@($seeds|Sort-Object -Unique)}; foreach($seed in $seeds){Write-Host "Testing seed: $seed"; python client.py --server http://localhost:8000 --seed $seed --text "[S1] Hello. Thank you for calling Inspira Financial. How can I help you today? [S2] Hello. I need assistance with my account." --output "seed_$seed.wav"}
