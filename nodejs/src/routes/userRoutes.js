const express = require("express");
const userController = require("../controllers/userController");

const router = express.Router();

router.post("/login", userController.login);
// router.get("/", userController.getAllUsers);
router.get("/badges", userController.getBadgesByUserCode);

module.exports = router;
