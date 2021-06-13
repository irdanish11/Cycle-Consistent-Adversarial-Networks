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
    #rm ${ZIP_FILE}   
    #Set the dataset as  per template
    mkdir -p "$TARGET_DIR/train" "$TARGET_DIR/test"
    mv "$TARGET_DIR/trainA" "$TARGET_DIR/train/A"
    mv "$TARGET_DIR/trainB" "$TARGET_DIR/train/B"
    mv "$TARGET_DIR/testA" "$TARGET_DIR/test/A"
    mv "$TARGET_DIR/testB" "$TARGET_DIR/test/B"
} 


mkdir data
cd data
echo '' 
echo 'Current Directory'
pwd
echo ''
array=("apple2orange" "summer2winter_yosemite" "horse2zebra" "monet2photo" 
       "cezanne2photo" "ukiyoe2photo" "vangogh2photo" "maps" "cityscapes" 
       "facades" "iphone2dslr_flower")
i=1
for list in "${array[@]}"
do
    echo "$i $list"
    let "i+=1" 
done
echo ''
read -p "Enter your choice: " ch
#downloading data
echo -e "\nDownloading data: ${array[1]}\n"
get_data "${array[1]}"


