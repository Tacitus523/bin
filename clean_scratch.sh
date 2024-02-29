#be01-be24
#bn01-bn04
#cn01-cn28
#nn01-nn08
#gtx01-gtx10

if ! command -v qstat &> /dev/null
then 
    echo "Error: Need to check the queue first, therefore need access to qstat"
    exit 1
fi

if [ `qstat | wc -l` -gt 0 ]
then
    echo "Error: Won't clean while job is running"
    exit 1
fi

delete_user_scratch_on_knecht () {
    if [ -d /scratch/$knecht/$USER ]
    then
        rm -r /scratch/$knecht/$USER/* &> /dev/null
    fi
}

for num in {01..24}
do
    knecht=be$num
    delete_user_scratch_on_knecht
done

for num in {01..04}
do
    knecht=bn$num
    delete_user_scratch_on_knecht
done

for num in {01..28}
do
    knecht=cn$num
    delete_user_scratch_on_knecht
done

for num in {01..08}
do
    knecht=nn$num
    delete_user_scratch_on_knecht
done

for num in {00..10}
do
    knecht=gtx$num
    delete_user_scratch_on_knecht
done

exit 0