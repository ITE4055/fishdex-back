const express = require("express");
const userController = require("../controllers/userController");

const router = express.Router();

router.post("/login", userController.login);
// r outer.get("/", userController.getAllUsers);

module.exports = router;
