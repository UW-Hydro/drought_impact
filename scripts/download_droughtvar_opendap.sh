met_url="http://thredds.northwestknowledge.net:8080/thredds/fileServer/MET"
met_var=("spi" "spei" "eddi")
time_intervals=("90d" "5y" "30d" "2y" "270d" "1y" "180d" "14d")

for met in ${met_var[*]}
do
	for time_interval in ${time_intervals[*]}
	do
		netcdf_url="${met_url}/${met}/${met}${time_interval}.nc"
		save_url="../data/drought_measures/${met}/${met}${time_interval}.nc"

		echo "Downloading ${met}${time_interval}"

		wget "$netcdf_url" -O "$save_url"
	done
done



