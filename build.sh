cd backend
sh build.sh

cd ../frontend
npm run build
rm -rf /Users/mars.zhu/code/azure/h5-game/assistantui/dist
cp -r dist /Users/mars.zhu/code/azure/h5-game/assistantui/dist

cd /Users/mars.zhu/code/azure/h5-game
sh build.sh
