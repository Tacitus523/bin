#!/bin/bash
#folder-prefix as $1, file-prefix as $2

ESPS_BY_MM_FILE=esps_by_mm.txt
ESP_GRADIENT_FILE=esp_gradients.txt

if [ -f $ESPS_BY_MM_FILE ]
then rm $ESPS_BY_MM_FILE
fi

if [ -f $ESP_GRADIENT_FILE ]
then rm $ESP_GRADIENT_FILE
fi

echo "Start ESP calculation `date`"
python3 /lustre/home/ka/ka_ipc/ka_he8978/bin/esp_calculation_from_pc.py --dir $1 --input $2 --unit V # ESP in Volt, change to au, if requiered
echo "End ESP calculation `date`"

source_dir=$PWD

for folder in $1*
    do 
    cd $folder
    cat esps_by_mm.txt >> $source_dir/$ESPS_BY_MM_FILE # first file name hardcoded like this in esp_calculation_from_pc.py
    cat esp_gradients.txt >> $source_dir/$ESP_GRADIENT_FILE # first file name hardcoded like this in esp_calculation_from_pc.py
    cd $source_dir
done
