methods=(
    const momentum ada_grad rms_prop ada_delta adam
)

rm -rf ../out/
mkdir ../out/

for method in ${methods[@]}
do
    python3 ${method}.py > ../out/${method}.txt
done