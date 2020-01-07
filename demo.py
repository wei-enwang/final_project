import wx
import os
app = wx.PySimpleApp()
wildcard = "All files (*.*)|*.*"
dialog = wx.FileDialog(None, "Choose a file", os.getcwd(), "", wildcard, wx.FD_OPEN)
if dialog.ShowModal() == wx.ID_OK:
    file1path=dialog.GetPath()
dialog.Destroy()
dialog = wx.FileDialog(None, "Choose a file", os.getcwd(), "", wildcard, wx.FD_OPEN)
if dialog.ShowModal() == wx.ID_OK:
    file2path=dialog.GetPath()
dialog.Destroy()
#print(file1path)
#print(file2path)
