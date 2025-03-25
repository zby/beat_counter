#!/bin/bash

sudo supervisorctl stop all
mv logs/* logs.back/
sudo supervisorctl start all
