import wx
import os
app = wx.PySimpleApp()
wildcard = " pic|*.jepg|pic|*.bmp|pic|*.gif|pic|*.jpg|pic|*.png"
print("please select a picture that include target pose:")
dialog = wx.FileDialog(None, "Choose a file", os.getcwd(), "", wildcard, wx.FD_OPEN)
if dialog.ShowModal() == wx.ID_OK:
    file1path=dialog.GetPath()
elif dialog.ShowModal()==wx.ID_CANCEL:
    print("haha")
dialog.Destroy()
print("please select a picture that include the person you want to perform the pose:")
dialog = wx.FileDialog(None, "Choose a file", os.getcwd(), "", wildcard, wx.FD_OPEN)
if dialog.ShowModal() == wx.ID_OK:
    file2path=dialog.GetPath()
elif dialog.ShowModal()==wx.ID_CANCEL:
    print("haha")
dialog.Destroy()
#print(file1path)
#print(file2path)
