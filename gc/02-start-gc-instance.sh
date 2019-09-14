#!/bin/bash
set -x #echo on

# Start instance
gcloud compute instances start --zone=$'us-central1-b' 'pytorch'