met_url="http://thredds.northwestknowledge.net:8080/thredds/fileServer/MET"
#met_var=("pr" "rmin" "rmax" "sph" "th" "vs" "tmmn" "tmmx" "bi")
#met_name=("precip" "rh_min" "rh_max" "sph" "wind_dir" "wind_spd" "tair_min" "tair_max" "burn_idx")
met_var=("th" "vs" "tmmn" "tmmx" "bi")
met_name=("wind_dir" "wind_spd" "tair_min" "tair_max" "burn_idx")

for i in "${!met_var[@]}"
do
    met=${met_var[i]}
    name=${met_name[i]}

    echo "Getting ${name} ..."

    save_dir="/pool0/home/steinadi/data/drought/drought_impact/data/met/${name}"
    #mkdir $save_dir

    for year in $(seq 1979 2022)
    do
        echo "  ... ${year}"

        netcdf_url="${met_url}/${met}/${met}_${year}.nc"
                
        save_path="${save_dir}/${name}_${year}.nc"

        #rm "$save_url"
        
        wget "$netcdf_url" -O "$save_path"
    done
done