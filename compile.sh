pyarmor-7 obfuscate -r interface.py
mv dist/ ../dist/
cp pm2_run.py ../dist/pm2_run.py
cp modules/rtsp_stream_processor.py ../dist/modules/rtsp_stream_processor.py
cp -r config/ ../dist/config/
cp -r templates/ ../dist/templates/
cp -r models/ ../dist/models/
mv ../dist/ ../FX/
