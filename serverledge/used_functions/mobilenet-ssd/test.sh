
CLI=$HOME/Dev/serverledge/bin/serverledge-cli

$CLI create -u -f mobilenetFunc --memory 900 --runtime custom --custom_image grussorusso/mobilenetssd
$CLI invoke -f mobilenetFunc --params_file input.json --ret_output