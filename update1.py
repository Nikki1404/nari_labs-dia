$seeds=@(); while($seeds.Count -lt 100){$seeds+=Get-Random -Minimum 100 -Maximum 50000;$seeds=@($seeds|Sort-Object -Unique)}; foreach($seed in $seeds){Write-Host "Testing seed: $seed"; python client.py --server http://localhost:8000 --seed $seed --text "[S1] Hello. Thank you for calling Inspira Financial. How can I help you today? [S2] Hello. I need assistance with my account. Could you please help me with that ?" --output "seed_$seed.wav"}

python client.py --server http://localhost:8000 --text-file full_call.txt --reference-audio reference.wav --reference-text reference.txt --max-new-tokens 4096 --output full_call.wav --play


================================================================================
DIA TTS STARTUP
================================================================================
Model             : nari-labs/Dia-1.6B-0626
Device            : cuda
PyTorch           : 2.6.0+cu124
PyTorch CUDA      : 12.4
CUDA available    : True
GPU               : NVIDIA A10G
DTYPE             : torch.float16
================================================================================
`torch_dtype` is deprecated! Use `dtype` instead!
Fetching 2 files: 100%|██████████| 2/2 [01:38<00:00, 49.10s/it]
ERROR:    Traceback (most recent call last):
