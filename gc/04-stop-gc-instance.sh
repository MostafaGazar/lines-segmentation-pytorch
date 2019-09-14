#!/bin/bash
set -x #echo on

gcloud compute instances stop --zone=$'us-central1-b' 'pytorch'