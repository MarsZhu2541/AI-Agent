cd backend
sh build.sh

cd ../frontend
npm run build
rm -rf /Users/mars.zhu/code/github/h5_game/assistantui/dist
cp -r dist /Users/mars.zhu/code/github/h5_game/assistantui/dist

cd /Users/mars.zhu/code/github/h5_game
sh build.sh
