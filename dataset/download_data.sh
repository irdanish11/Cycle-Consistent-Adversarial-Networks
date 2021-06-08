get_data () {
    URL=https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/"$1".zip
    #echo $URL
    ZIP_FILE="$1".zip
    TARGET_DIR="$1"
    #downloading data
    echo "Downloading the dataset: $TARGET_DIR"
    wget ${URL}
    #unziping
    unzip ${ZIP_FILE}
    rm ${ZIP_FILE}   
    #Set the dataset as  per template
    mkdir -p "data/$TARGET_DIR/train" "data/$TARGET_DIR/test"
    mv "data/$TARGET_DIR/trainA" "data/$TARGET_DIR/train/A"
    mv "data/$TARGET_DIR/trainB" "data/$TARGET_DIR/train/B"
    mv "data/$TARGET_DIR/testA" "data/$TARGET_DIR/test/A"
    mv "data/$TARGET_DIR/testB" "data/$TARGET_DIR/test/B"
} 


array=("apple2orange" "summer2winter_yosemite" "horse2zebra" "monet2photo" 
       "cezanne2photo" "ukiyoe2photo" "vangogh2photo" "maps" "cityscapes" 
       "facades" "iphone2dslr_flower")

for list in "${array[@]}"
do
    for item in $list
    do
        get_data "$item"
    done
done


