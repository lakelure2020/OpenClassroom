How to start the script/Git commands to push your change to the repo:

Navigate to your local repository on your computer using the command line. You can do this with the cd (change directory) command. For example:
cd /path/to/your/repository

Stage the file you want to push. This means adding the file to a list of files that you want to commit to the repository. Use the git add command followed by the name of the file:
git add filename

If you want to add all files in the directory, you can use . in place of filename.

Commit the file. This saves a snapshot of the file in your local repository. Use the git commit command followed by a message describing what changes you made:
git commit -m "Your message about the commit"

Push the commit to the remote repository. This updates the remote repository with any commits made locally that aren’t already in the remote repo. Use the git push command:
git push origin master

Here, origin is the default name Git gives to the server where your repo is stored, and master is the branch you’re pushing to.

Remember to replace /path/to/your/repository, filename, and "Your message about the commit" with your actual directory path, file name, and commit message. Also, make sure you have the necessary permissions to push to the repository. If the repository is not initialized yet, you might need to initialize it using git init command. If the remote is not set, you might need to add it using git remote add origin yourRemoteUrl. If you’re working on a different branch, replace master with your branch name.

