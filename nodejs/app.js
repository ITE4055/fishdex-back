const express = require("express");
const config = require("./src/config");
const routes = require("./src/routes");

const app = express();

// 미들웨어 설정
app.use(express.json());

// 라우터 설정
app.use("/api", routes);

module.exports = app;
