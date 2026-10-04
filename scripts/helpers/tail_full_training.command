#!/bin/zsh
cd /Users/sachin/h3-avatar-aq || exit 1
echo "Tailing Modal logs for full H3 Ref2VA LoRA training..."
echo "Run: ap-Mk5JAjytLfnaYobhIhrf1e"
echo "Output: /runs/h3/ref2va-lora-full-8000"
echo
modal app logs ap-Mk5JAjytLfnaYobhIhrf1e -f --timestamps
echo
echo "Modal log stream ended. Press return to close this window."
read
